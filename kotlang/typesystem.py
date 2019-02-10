from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast, List, Optional, Tuple, TYPE_CHECKING

from llvmlite import ir


@dataclass
class Type:
    def get_ir_type(self) -> ir.Type:
        raise NotImplementedError(f'Not implemented for {type(self)}')

    @property
    def name(self) -> str:
        raise NotImplementedError(f'Not implemented for {type(self)}')

    def as_pointer(self) -> PointerType:
        return PointerType(self)

    def as_pointee(self) -> Type:
        assert isinstance(self, PointerType)
        return self.pointee

    def __eq__(self, other: Any) -> bool:
        return hasattr(other, 'name') and self.name == other.name

    def adapt(self, builder: ir.IRBuilder, value: ir.Value, from_type: Type) -> ir.Value:
        assert self == from_type, f'Cannot adapt {from_type.name} to {self.name}'
        return value

    def alignment(self) -> Optional[int]:
        return None

    def get_required_size_in_bytes(self) -> int:
        raise NotImplementedError(type(self).__name__)


@dataclass
class VoidType(Type):
    @property
    def name(self) -> str:
        return 'void'

    def get_ir_type(self) -> ir.Type:
        return ir.VoidType()


void = VoidType()


@dataclass
class IntType(Type):
    bits: int
    signed: bool

    @property
    def name(self) -> str:
        prefix = 'i' if self.signed else 'u'
        return f'{prefix}{self.bits}'

    def get_ir_type(self) -> ir.Type:
        return ir.IntType(self.bits)

    def adapt(self, builder: ir.IRBuilder, value: ir.Value, from_type: Type) -> ir.Value:
        if not isinstance(from_type, IntType):
            return super().adapt(builder, value, from_type)

        if from_type.bits == self.bits:
            assert from_type.signed == self.signed
            return value

        ir_type = self.get_ir_type()
        if from_type.bits > self.bits:
            return builder.trunc(value, ir_type)

        if from_type.signed:
            assert self.signed
            return builder.sext(value, ir_type)
        else:
            assert not self.signed
            return builder.zext(value, ir_type)

    def get_required_size_in_bytes(self) -> int:
        return cast(int, self.bits / 8)


@dataclass
class FloatType(Type):
    bits: int

    @property
    def name(self) -> str:
        return f'f{self.bits}'

    def get_ir_type(self) -> ir.Type:
        return {32: ir.FloatType(), 64: ir.DoubleType(), 80: IRLongDoubleType()}[self.bits]


class IRLongDoubleType(ir.types._BaseFloatType):  # type: ignore
    # So the story of this class is llvmlite only exposes float (32-bit) and double (64-bit) types.
    # On my Mac when I use "long double" clang uses x86_fp80 type in the default configuration so
    # let's use it.
    # TODO: See if this is actually correct.
    null = '0.0'
    intrinsic_name = 'x86_fp80'

    def __str__(self) -> str:
        return self.intrinsic_name

    def format_constant(self, value: float) -> str:
        # TODO: is _format_double enough here?
        return cast(str, ir.types._format_double(value))


# llvmlite does this internally for the builtin types, we follow suit.
IRLongDoubleType._create_instance()


@dataclass
class BoolType(Type):
    @property
    def name(self) -> str:
        return 'bool'

    def get_ir_type(self) -> ir.Type:
        return ir.IntType(1)


@dataclass
class PointerType(Type):
    pointee: Type

    @property
    def name(self) -> str:
        return self.pointee.name + '*'

    def get_ir_type(self) -> ir.Type:
        return self.pointee.get_ir_type().as_pointer()

    def adapt(self, builder: ir.IRBuilder, value: ir.Value, from_type: Type) -> ir.Value:
        # TODO remove this
        i64 = ir.IntType(64)

        if isinstance(from_type, ArrayType) and self.pointee == from_type.element_type:
            memory = builder.alloca(value.type)
            builder.store(value, memory)
            return builder.gep(memory, (i64(0), i64(0)))

        return super().adapt(builder, value, from_type)

    def get_required_size_in_bytes(self) -> int:
        # HACK: x86 64-bit-specific
        return 8


class DottableType(Type):
    def get_member_type(self, name: str) -> Type:
        raise NotImplementedError()

    def get_member_pointer(self, builder: ir.IRBuilder, base_pointer: ir.Value, name: str) -> ir.Value:
        raise NotImplementedError()


@dataclass
class StructType(DottableType):
    struct_name: str
    members: List[Tuple[str, Type]]

    @property
    def name(self) -> str:
        return self.struct_name

    def get_ir_type(self) -> ir.Type:
        member_types = [t.get_ir_type() for n, t in self.members]
        return ir.LiteralStructType(member_types)

    def get_member_pointer(self, builder: ir.IRBuilder, base_pointer: ir.Value, name: str) -> ir.Value:
        member_index = -1
        for i, (n, t) in enumerate(self.members):
            if n == name:
                member_index = i
        if member_index == -1:
            raise KeyError()

        # i32 is mandatory when indexing within a structure.
        # See http://llvm.org/docs/LangRef.html#getelementptr-instruction
        i32 = ir.IntType(32)
        i64 = ir.IntType(64)
        return builder.gep(base_pointer, (i64(0), i32(member_index)))

    def get_member_type(self, name: str) -> Type:
        for n, t in self.members:
            if n == name:
                return t
        raise KeyError()


@dataclass
class UnionType(DottableType):
    union_name: str
    members: List[Tuple[str, Type]]

    @property
    def name(self) -> str:
        return self.union_name

    def get_ir_type(self) -> ir.Type:
        size = max(t.get_required_size_in_bytes() for (_, t) in self.members)
        return ir.LiteralStructType([ir.ArrayType(ir.IntType(8), size)])

    def get_member_pointer(self, builder: ir.IRBuilder, base_pointer: ir.Value, name: str) -> ir.Value:
        member_type = self.get_member_type(name)
        member_ir_type = member_type.get_ir_type()
        return builder.bitcast(base_pointer, member_ir_type.as_pointer())

    def get_member_type(self, name: str) -> Type:
        for n, t in self.members:
            if n == name:
                return t
        raise KeyError()

    def alignment(self) -> int:
        # TODO: deduce this
        return 8


@dataclass
class ArrayType(Type):
    element_type: Type
    length: int

    @property
    def name(self) -> str:
        return f'{self.element_type.name}[{self.length}]'

    def get_ir_type(self) -> ir.Type:
        return ir.ArrayType(self.element_type.get_ir_type(), self.length)

    def get_required_size_in_bytes(self) -> int:
        return self.length * self.element_type.get_required_size_in_bytes()


@dataclass
class FunctionType(Type):
    parameter_types: List[Type]
    return_type: Type
    variadic: bool

    def get_ir_type(self) -> ir.Type:
        return ir.FunctionType(
            self.return_type.get_ir_type(), [t.get_ir_type() for t in self.parameter_types], self.variadic
        )
